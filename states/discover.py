"""
LLM-based state discovery: groups log templates into semantic states
using observed transitions and trace samples as context.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

from config import MAX_ITERATIONS, MAX_TEMPLATES, MAX_TRACE_LENGTH, MAX_TRACES

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a State Discovery Agent for log analysis.

Given a list of log templates, their observed transitions, and sample trace sequences,
group templates into semantic states. A state represents a logical phase or module in
the system's execution.

Rules:
1. Use observed transitions: templates that frequently follow each other likely belong to
   the same state or adjacent states in a workflow.
2. Group templates from the same module together (e.g., all DataTransfer -> S_DATA_TRANSFER).
3. Group templates from the same execution phase (e.g., all init -> S_INIT).
4. Use 3-12 states total -- not too few, not too many.
5. Every template must be mapped exactly once. No omissions.
6. Use descriptive state IDs: S_INIT, S_DATA_TRANSFER, S_FILE_WRITE, S_SHUTDOWN, etc.

Return STRICT JSON with this shape:
{
  "states": [
    {"state_id": "S_INIT", "state_name": "Initialization", "state_desc": "..."}
  ],
  "template_state_map": [
    {"template_id": "TMP_ABCD", "state_id": "S_INIT"}
  ]
}

Do NOT include any text outside the JSON object."""


def _parse_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError as original_error:
        parse_error = original_error

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(content[start : end + 1])

    raise parse_error


def _validate(state_map: dict[str, str], required: set[str]) -> dict[str, str]:
    missing = required - set(state_map.keys())
    if missing:
        logger.warning("LLM missed %d templates; falling back to S_UNMAPPED", len(missing))
        for tid in missing:
            state_map[tid] = "S_UNMAPPED"
    return state_map


def _score(state_map: dict[str, str], required: set[str]) -> float:
    mapped = set(state_map.keys())
    coverage = len(mapped & required) / max(len(required), 1)
    states = set(state_map.values())
    n_states = len(states)
    if n_states < 3:
        coverage -= 0.2
    elif n_states > 12:
        coverage -= 0.1
    state_counts = Counter(state_map.values())
    if state_counts and max(state_counts.values()) / len(state_map) > 0.65:
        coverage -= 0.1
    return coverage


def _build_feedback(required: set[str], state_map: dict[str, str]) -> dict:
    missing = sorted(required - set(state_map.keys()))
    state_counts = Counter(state_map.values())
    largest = max(state_counts.values()) / max(len(state_map), 1)
    feedback: dict = {"status": "incomplete", "missing_template_ids": missing[:20]}
    if largest > 0.65:
        feedback["warning"] = (
            f"Largest state contains {largest:.1%} of templates. "
            "Split catch-all states into finer-grained states."
        )
    feedback["instruction"] = (
        "Map ALL templates. Ensure every template_id appears exactly once. "
        "Use 3-12 states. Split catch-all states, merge fragmented states."
    )
    return feedback


def discover_states(
    parsed_events_csv: str | Path,
    llm_client,
    max_templates: int = MAX_TEMPLATES,
    max_traces: int = MAX_TRACES,
    max_trace_length: int = MAX_TRACE_LENGTH,
    max_iterations: int = MAX_ITERATIONS,
) -> dict[str, str]:
    templates: dict[str, str] = {}
    template_counts: Counter[str] = Counter()
    traces: dict[str, list[dict]] = defaultdict(list)

    with open(parsed_events_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = row.get("template_id", "")
            text = row.get("template_text", "")
            if tid and tid not in templates:
                templates[tid] = text
            if tid:
                template_counts[tid] += 1
            traces[row.get("trace_id", "")].append(row)

    selected = [tid for tid, _ in template_counts.most_common(max_templates)]
    required = set(selected)
    selected_set = set(selected)

    template_list = [
        {"template_id": tid, "template_text": templates[tid], "count": template_counts[tid]}
        for tid in selected
    ]

    transition_counts: Counter[tuple[str, str]] = Counter()
    trace_samples: list[dict] = []

    for trace_id, trace_events in list(traces.items())[:max_traces]:
        ordered = sorted(trace_events, key=lambda e: int(e.get("line_number") or 0))
        sequence = [
            e["template_id"] for e in ordered
            if e.get("template_id") in selected_set
        ][:max_trace_length]
        if sequence:
            trace_samples.append({"trace_id": trace_id, "template_sequence": sequence})
        for src, dst in zip(sequence, sequence[1:]):
            transition_counts[(src, dst)] += 1

    observed_transitions = [
        {"src_template": s, "dst_template": d, "count": c}
        for (s, d), c in transition_counts.most_common(300)
    ]

    user_payload = {
        "templates": template_list,
        "total_templates": len(templates),
        "observed_template_transitions": observed_transitions,
        "trace_samples": trace_samples,
        "instruction": (
            "Group these templates into semantic states (3-12 states). "
            "Use observed transitions and trace samples to infer which templates "
            "belong together."
        ),
    }

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, indent=2, ensure_ascii=False)},
    ]

    feedback: dict | None = None
    best_result: dict[str, str] | None = None
    best_score = -1.0

    for iteration in range(1, max_iterations + 1):
        logger.info("State discovery iteration %d/%d", iteration, max_iterations)

        if feedback:
            messages.append({
                "role": "user",
                "content": json.dumps(feedback, indent=2, ensure_ascii=False),
            })

        try:
            for attempt in range(1, 4):
                try:
                    response = llm_client.chat(messages)
                    result = _parse_json(response)
                    break
                except json.JSONDecodeError as exc:
                    if attempt == 3:
                        raise
                    logger.warning("JSON parse attempt %d failed: %s", attempt, exc)
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON. "
                            "Return only a single valid JSON object. "
                            "No markdown, no code fences, no extra text.\n"
                            f"Error: {exc}"
                        ),
                    })

            state_map = _validate(result, required)
            score = _score(state_map, required)

            n_states = len(set(state_map.values()))
            coverage = len(set(state_map) & required) / max(len(required), 1)
            logger.info("Iteration %d: %d states, coverage=%.3f, score=%.3f",
                        iteration, n_states, coverage, score)

            if score > best_score:
                best_score = score
                best_result = state_map

            if score >= 1.0:
                return state_map

            feedback = _build_feedback(required, state_map)

        except Exception as exc:
            logger.warning("Iteration %d failed: %s", iteration, exc)
            feedback = {"error": str(exc), "instruction": "Return valid JSON with all templates mapped."}

    if best_result is None:
        raise ValueError("LLM did not produce any valid state mapping.")

    logger.warning("Returning best candidate (score=%.3f)", best_score)
    return best_result


def save_state_map(state_map: dict[str, str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["template_id", "state_id"])
        writer.writeheader()
        for tid in sorted(state_map.keys()):
            writer.writerow({"template_id": tid, "state_id": state_map[tid]})