# Zero-Shot Log Anomaly Detection via LLM-Guided State Discovery

Detect anomalies in system logs with **no labeled anomalies** — an LLM groups log
templates into semantic states, and per-trace state-transition graphs are scored
on structural dimensions (entropy, complexity, density, dominance).

A companion supervised study (also in this repo) shows that for highly regular
systems, **order beats frequency**: a first-order Markov model reaches 82.3%
accuracy vs. 61.1% for Bag-of-Templates + XGBoost (+21.2pp).

## Architecture

```
raw .log files
    │
    ▼ (optional — needs the LogMine binary)
┌───────────┐   cluster    ┌──────────────────┐  convert   ┌─────────────────────┐
│  cluster  │ ──────────▶ │ *_clustered.log  │ ──────────▶ │ *_parsed_events.csv │
└───────────┘              └──────────────────┘              └─────────────────────┘
                                                                   │
                          ┌──────────────────┐   discover          ▼
                          │ LLM state map    │ ◀────────  template_state_map.csv
                          │ (3–12 states)    │   (needs OPENAI_API_KEY)
                          └──────────────────┘
                                   │
                                   ▼
                          ┌──────────────────┐    score    ┌──────────────────┐
                          │ state-transition │ ──────────▶ │ trace scores +   │
                          │ graphs (JSONL)   │             │ line-level report│
                          └──────────────────┘             └──────────────────┘
                                   │
                                   │ plot
                                   ▼
                          ┌──────────────────┐
                          │ networkx figures │
                          └──────────────────┘
```

## Results (supervised track, `supervised.py`)

20-event windows over 14 days of production-style logs, 15% injected anomalies
(shuffle / repeat / insert / drop):

| Method              | Accuracy | Precision | Recall | F1   | ROC-AUC |
|---------------------|----------|-----------|--------|------|---------|
| Bag-of-Templates + XGBoost | 0.611 | 0.22 | 0.63 | 0.33 | 0.69 |
| Bigram + XGBoost    | 0.612 | 0.23 | **0.70** | 0.35 | 0.75 |
| **Markov (1st-order)** | **0.823** | 0.42 | 0.45 | 0.43 | 0.75 |
| Ensemble (avg)      | 0.879 | 0.95 | 0.20 | 0.33 | 0.80 |

Takeaway: BoT is permutation-invariant and blind to order-based anomalies;
sequence-aware methods win. Markov needs no labeled anomalies — only normal data.
See `paper/main.tex` for the full analysis and `paper/llm_paper.tex` for the
zero-shot pipeline paper.

## Quickstart

```bash
pip install -r requirements.txt   # xgboost only needed for the supervised track

# 1. Generate synthetic demo logs (no proprietary data needed)
python scripts/generate_synthetic_logs.py --days 5 --sessions-per-day 40

# 2. Convert clustered logs -> structured events
#    (clustered files are generated for you; the `cluster` step needs the
#     external LogMine binary and can be skipped for the demo)
python pipeline.py convert

# 3. LLM state discovery (any OpenAI-compatible endpoint)
export OPENAI_API_KEY=sk-...
# export LLM_BASE_URL=https://api.openai.com/v1   # default
# export LLM_MODEL=gpt-4o-mini                    # default
python pipeline.py discover

# 4. Score traces + line-level anomalies
python pipeline.py score

# 5. Visualize the top trace
python pipeline.py plot data/graphs/synth_2026-01-22_graphs.jsonl --output top_trace.png

# Supervised benchmark (no LLM needed)
python supervised.py --csv-dir data/processed
```

Other commands: `python pipeline.py all` (cluster→convert→discover→score),
`--force` to redo state discovery, `--model` / `--max-templates` to tune it,
`--report-name` for the score report.

## Requirements

- Python 3.10+
- numpy, scikit-learn, matplotlib, networkx
- xgboost — optional, only for `supervised.py`
- LogMine binary — optional, only for `pipeline.py cluster` on raw logs
  (set `LOGMINE_BIN` or place it where `config.py` expects it)

## Project structure

```
.
├── pipeline.py                 # entry point: cluster → convert → discover → score → plot
├── config.py                   # paths + LLM config, all env-overridable
├── supervised.py               # BoT/XGBoost vs bigram/XGBoost vs Markov benchmark
├── preprocess/                 # LogMine wrapper + cluster→events converter
├── states/                     # LLM state discovery (stdlib-only OpenAI client)
├── detect/                     # graph builder, 4-dimension scorer, evaluator
├── visualize/                  # networkx state-transition plots
├── scripts/
│   └── generate_synthetic_logs.py  # demo data generator (no real logs needed)
└── paper/
    ├── main.tex                # "The Limits of Counting" (supervised study)
    ├── llm_paper.tex           # "Zero-Shot Log Anomaly Detection via LLM-Guided Semantic State Discovery"
    └── fig_*.pdf               # paper figures
```

## Notes

- This repo was sanitized for public release: the original proprietary
  production logs were removed, and the demo runs entirely on synthetic logs
  from `scripts/generate_synthetic_logs.py`. Reported paper numbers were
  produced on the original dataset with synthetically injected anomalies
  (see the Limitations section of `paper/main.tex`).
- LLM endpoint, model, and key are configured via `LLM_BASE_URL`, `LLM_MODEL`,
  and `OPENAI_API_KEY` — never hardcode credentials.
- `data/` is gitignored; generated artifacts stay local.
