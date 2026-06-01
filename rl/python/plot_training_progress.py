#!/usr/bin/env python3
"""
Plot or summarize masked PPO JSONL training progress.

Run from the cyber-sim-engine repo root after training with --progress-path:

    py rl/python/plot_training_progress.py \
      --progress rl/progress/masked_ppo.jsonl \
      --out-dir rl/plots

If matplotlib is unavailable, the script still writes a compact CSV summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


METRIC_FIELDS = [
    "episode",
    "recentAvgReward",
    "trainWinRate",
    "evalDetWinRate",
    "evalStoWinRate",
    "loss",
    "policyLoss",
    "valueLoss",
    "entropy",
    "approxKl",
    "clipFrac",
    "ppoUpdates",
    "policyDecisions",
]


def default_engine_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_value: str, engine_root: Path) -> Path:
    p = Path(path_value)
    if not p.is_absolute():
        p = engine_root / p
    return p.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot JSONL progress produced by train_ppo_masked.py --progress-path.")
    parser.add_argument("--engine-root", default=str(default_engine_root()), help="Path to cyber-sim-engine repo root.")
    parser.add_argument("--progress", default="rl/progress/masked_ppo.jsonl", help="Progress JSONL file, relative to engine root unless absolute.")
    parser.add_argument("--out-dir", default="rl/plots", help="Output directory, relative to engine root unless absolute.")
    parser.add_argument("--prefix", default="masked_ppo", help="Filename prefix for generated files.")
    parser.add_argument("--no-plots", action="store_true", help="Only write CSV and summary JSON.")
    return parser.parse_args()


def _safe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _read_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if isinstance(event, dict):
                events.append(event)
    return events


def _flatten_event(event: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {k: None for k in METRIC_FIELDS}
    row["episode"] = event.get("episode")
    row["recentAvgReward"] = event.get("recentAvgReward")
    row["trainWinRate"] = event.get("trainWinRate")
    row["ppoUpdates"] = event.get("ppoUpdates")

    update = event.get("update") or event.get("lastUpdate")
    if isinstance(update, dict):
        row["loss"] = update.get("loss")
        row["policyLoss"] = update.get("policyLoss")
        row["valueLoss"] = update.get("valueLoss")
        row["entropy"] = update.get("entropy")
        row["approxKl"] = update.get("approxKl")
        row["clipFrac"] = update.get("clipFrac")
        row["policyDecisions"] = update.get("policyDecisions")

    eval_stats = event.get("eval")
    if isinstance(eval_stats, dict):
        det = eval_stats.get("deterministic")
        sto = eval_stats.get("stochastic")
        if isinstance(det, dict):
            row["evalDetWinRate"] = det.get("winRate")
        if isinstance(sto, dict):
            row["evalStoWinRate"] = sto.get("winRate")

    return row


def _rows_from_events(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    last_by_field: Dict[str, Any] = {}
    for event in events:
        row = _flatten_event(event)
        if row.get("episode") is None:
            continue
        # Carry forward slow-changing metrics so plots have continuous lines.
        for field in METRIC_FIELDS:
            if field == "episode":
                continue
            if row.get(field) is None and field in last_by_field:
                row[field] = last_by_field[field]
            elif row.get(field) is not None:
                last_by_field[field] = row[field]
        rows.append(row)
    rows.sort(key=lambda r: int(r.get("episode") or 0))
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in METRIC_FIELDS})


def _series(rows: List[Dict[str, Any]], field: str) -> tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    for row in rows:
        episode = row.get("episode")
        value = _safe_float(row.get(field))
        if episode is None or value is None:
            continue
        xs.append(int(episode))
        ys.append(value)
    return xs, ys


def _plot_metric(rows: List[Dict[str, Any]], field: str, out_path: Path, title: str, ylabel: str) -> Optional[str]:
    xs, ys = _series(rows, field)
    if not xs:
        return None
    import matplotlib.pyplot as plt  # noqa: WPS433

    plt.figure()
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return str(out_path)


def _write_plots(rows: List[Dict[str, Any]], out_dir: Path, prefix: str) -> List[str]:
    plots: List[str] = []
    specs = [
        ("recentAvgReward", "Recent average reward", "Reward"),
        ("trainWinRate", "Training win rate", "Win rate"),
        ("evalDetWinRate", "Deterministic eval win rate", "Win rate"),
        ("evalStoWinRate", "Stochastic eval win rate", "Win rate"),
        ("loss", "PPO loss", "Loss"),
        ("policyLoss", "Policy loss", "Loss"),
        ("valueLoss", "Value loss", "Loss"),
        ("entropy", "Policy entropy", "Entropy"),
        ("approxKl", "Approximate KL", "KL"),
        ("clipFrac", "Clip fraction", "Fraction"),
    ]
    for field, title, ylabel in specs:
        result = _plot_metric(rows, field, out_dir / f"{prefix}_{field}.png", title, ylabel)
        if result:
            plots.append(result)
    return plots


def _summary(events: List[Dict[str, Any]], rows: List[Dict[str, Any]], csv_path: Path, plots: List[str]) -> Dict[str, Any]:
    final_events = [e for e in events if e.get("event") == "final"]
    final_event = final_events[-1] if final_events else None
    last_row = rows[-1] if rows else {}
    return {
        "ok": True,
        "events": len(events),
        "rows": len(rows),
        "firstEpisode": rows[0].get("episode") if rows else None,
        "lastEpisode": rows[-1].get("episode") if rows else None,
        "lastTrainWinRate": last_row.get("trainWinRate"),
        "lastEvalDetWinRate": last_row.get("evalDetWinRate"),
        "lastEvalStoWinRate": last_row.get("evalStoWinRate"),
        "lastLoss": last_row.get("loss"),
        "final": final_event,
        "csv": str(csv_path),
        "plots": plots,
    }


def main() -> int:
    args = parse_args()
    engine_root = Path(args.engine_root).resolve()
    progress_path = resolve_path(args.progress, engine_root)
    out_dir = resolve_path(args.out_dir, engine_root)

    if not progress_path.exists():
        print(json.dumps({
            "ok": False,
            "error": f"Progress file not found: {progress_path}",
            "hint": "Train with --progress-path rl/progress/masked_ppo.jsonl first.",
        }, indent=2))
        return 1

    events = _read_events(progress_path)
    rows = _rows_from_events(events)
    csv_path = out_dir / f"{args.prefix}_progress.csv"
    _write_csv(csv_path, rows)

    plots: List[str] = []
    if not args.no_plots:
        try:
            plots = _write_plots(rows, out_dir, args.prefix)
        except Exception as exc:
            plots = []
            (out_dir / f"{args.prefix}_plot_error.txt").write_text(str(exc), encoding="utf-8")

    summary = _summary(events, rows, csv_path, plots)
    summary_path = out_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
