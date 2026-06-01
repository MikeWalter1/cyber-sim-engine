#!/usr/bin/env python3
"""
Run a stronger PPO evaluation suite.

This script evaluates the same PPO checkpoint in deterministic and stochastic
modes, against random and heuristic baselines, from both seats.

Run from the cyber-sim-engine repo root:

    py rl/python/evaluate_ppo_suite.py \
      --checkpoint rl/checkpoints/masked_ppo_best.pt \
      --deck1 decks/DEV-TEST-001.deck \
      --deck2 decks/DEV-TEST-002.deck \
      --episodes 1000 \
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_MATCHUPS = "policy_vs_random,random_vs_policy,policy_vs_heuristic,heuristic_vs_policy"


def default_engine_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_value: str, engine_root: Path) -> str:
    p = Path(path_value)
    if not p.is_absolute():
        p = engine_root / p
    return str(p.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a masked PPO checkpoint in deterministic and stochastic modes.")
    parser.add_argument("--engine-root", default=str(default_engine_root()), help="Path to cyber-sim-engine repo root.")
    parser.add_argument("--deck1", default="decks/DEV-TEST-001.deck", help="Deck 1 path, relative to engine root unless absolute.")
    parser.add_argument("--deck2", default="decks/DEV-TEST-002.deck", help="Deck 2 path, relative to engine root unless absolute.")
    parser.add_argument("--checkpoint", default="rl/checkpoints/masked_ppo_best.pt", help="PPO checkpoint path, relative to engine root unless absolute.")
    parser.add_argument("--node", default="node", help="Node executable path.")
    parser.add_argument("--first", default=None, choices=[None, "p1", "p2"], help="Force first player.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    parser.add_argument("--episodes", type=int, default=1000, help="Episodes per matchup per mode.")
    parser.add_argument("--turn-cap", type=int, default=200, help="Engine turn cap.")
    parser.add_argument("--max-actions", type=int, default=128, help="Fixed action mask size.")
    parser.add_argument("--max-steps-per-episode", type=int, default=5000, help="Python-side safety step cap.")
    parser.add_argument("--max-main-actions-per-turn", type=int, default=12, help="Force end turn after this many non-end main actions per player-turn.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--matchups", default=DEFAULT_MATCHUPS, help="Comma-separated matchup names, or all.")
    parser.add_argument("--modes", default="deterministic,stochastic", help="Comma-separated modes: deterministic,stochastic.")
    parser.add_argument("--output", default=None, help="Optional JSON output file, relative to engine root unless absolute.")
    parser.add_argument("--verbose", action="store_true", help="Print each underlying evaluator command to stderr.")
    return parser.parse_args()


def _run_one_mode(args: argparse.Namespace, engine_root: Path, mode: str) -> Dict[str, Any]:
    script = engine_root / "rl" / "python" / "evaluate_ppo.py"
    cmd: List[str] = [
        sys.executable,
        str(script),
        "--engine-root", str(engine_root),
        "--deck1", resolve_path(args.deck1, engine_root),
        "--deck2", resolve_path(args.deck2, engine_root),
        "--checkpoint", resolve_path(args.checkpoint, engine_root),
        "--node", args.node,
        "--seed", str(args.seed),
        "--episodes", str(args.episodes),
        "--turn-cap", str(args.turn_cap),
        "--max-actions", str(args.max_actions),
        "--max-steps-per-episode", str(args.max_steps_per_episode),
        "--max-main-actions-per-turn", str(args.max_main_actions_per_turn),
        "--device", args.device,
        "--matchups", args.matchups,
    ]
    if args.first:
        cmd.extend(["--first", args.first])
    if mode == "stochastic":
        cmd.append("--stochastic")
    elif mode != "deterministic":
        raise ValueError(f"Unknown evaluation mode: {mode!r}")

    if args.verbose:
        print(" ".join(cmd), file=sys.stderr, flush=True)

    proc = subprocess.run(cmd, cwd=str(engine_root), text=True, capture_output=True)
    if proc.returncode != 0:
        return {
            "ok": False,
            "mode": mode,
            "returnCode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "mode": mode,
            "error": f"Evaluator did not return JSON: {exc}",
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    data["mode"] = mode
    return data


def _summarize_mode(result: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": bool(result.get("ok")), "mode": result.get("mode"), "matchups": {}}
    for name, item in (result.get("matchups") or {}).items():
        if not isinstance(item, dict):
            continue
        p1_agent = item.get("p1Agent")
        p2_agent = item.get("p2Agent")
        policy_wins: Optional[int] = None
        policy_win_rate: Optional[float] = None
        if p1_agent == "policy":
            policy_wins = int(item.get("p1Wins") or 0)
            policy_win_rate = float(item.get("p1WinRate") or 0.0)
        elif p2_agent == "policy":
            policy_wins = int(item.get("p2Wins") or 0)
            policy_win_rate = float(item.get("p2WinRate") or 0.0)
        out["matchups"][name] = {
            "ok": item.get("ok"),
            "policyWinRate": policy_win_rate,
            "policyWins": policy_wins,
            "p1Wins": item.get("p1Wins"),
            "p2Wins": item.get("p2Wins"),
            "noWinner": item.get("noWinner"),
            "errors": item.get("errors"),
            "badObservations": item.get("badObservations"),
            "badActionMasks": item.get("badActionMasks"),
            "avgSteps": item.get("avgSteps"),
            "avgTurns": item.get("avgTurns"),
        }
    return out


def main() -> int:
    args = parse_args()
    engine_root = Path(args.engine_root).resolve()
    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    if not modes:
        raise SystemExit("No evaluation modes selected")

    results: Dict[str, Any] = {
        "ok": True,
        "checkpoint": resolve_path(args.checkpoint, engine_root),
        "episodesPerMatchup": args.episodes,
        "seed": args.seed,
        "matchups": args.matchups,
        "modes": {},
        "summary": {},
    }

    for mode in modes:
        mode_result = _run_one_mode(args, engine_root, mode)
        results["modes"][mode] = mode_result
        results["summary"][mode] = _summarize_mode(mode_result)
        if not mode_result.get("ok"):
            results["ok"] = False

    output_path = None
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = engine_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        results["output"] = str(output_path.resolve())

    print(json.dumps(results, indent=2))
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
