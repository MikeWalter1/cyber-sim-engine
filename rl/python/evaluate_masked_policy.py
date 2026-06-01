#!/usr/bin/env python3
"""Evaluate a saved Step 6 masked policy checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import torch
except Exception as exc:  # pragma: no cover
    print(json.dumps({
        "ok": False,
        "error": "PyTorch is required for Step 6 evaluation.",
        "install": "py -m pip install torch numpy",
        "detail": str(exc),
    }, indent=2))
    raise SystemExit(1)

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from agents import BaseAgent, make_agent, find_action_index  # noqa: E402
from cyber_env import CyberSimGymEnv  # noqa: E402
from masked_policy import TorchPolicyAgent, load_checkpoint  # noqa: E402


DEFAULT_MATCHUPS = [
    ("policy_vs_random", "policy", "random"),
    ("random_vs_policy", "random", "policy"),
    ("policy_vs_heuristic", "policy", "heuristic"),
    ("heuristic_vs_policy", "heuristic", "policy"),
]


def default_engine_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_value: str, engine_root: Path) -> str:
    p = Path(path_value)
    if not p.is_absolute():
        p = engine_root / p
    return str(p.resolve())


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    try:
        return list(value)
    except Exception:
        return []


def all_finite(values: Iterable[Any]) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in values)
    except Exception:
        return False


def validate_observation(obs: Any, expected_size: Optional[int]) -> Optional[str]:
    values = as_list(obs)
    if expected_size is not None and len(values) != expected_size:
        return f"observation length {len(values)} != expected {expected_size}"
    if not all_finite(values):
        return "observation contains non-finite values"
    return None


def validate_action_mask(info: Dict[str, Any], max_actions: int) -> Optional[str]:
    mask = info.get("action_mask")
    if not isinstance(mask, list):
        return "info['action_mask'] is missing or not a list"
    if len(mask) != max_actions:
        return f"action_mask length {len(mask)} != max_actions {max_actions}"
    try:
        active = sum(int(v) for v in mask)
        if any(int(v) not in (0, 1) for v in mask):
            return "action_mask contains non-binary entries"
    except Exception:
        return "action_mask contains non-integer-like entries"
    if active <= 0 and not info.get("winner"):
        return "action_mask has no legal actions before terminal state"
    if bool(info.get("actionMaskOverflow")):
        return "actionMaskOverflow is true"
    return None


def force_end_turn_if_needed(info: Dict[str, Any], main_action_counts: Dict[str, int], max_main_actions_per_turn: int) -> Optional[int]:
    if info.get("waitingForStep") != "main_phase":
        return None
    key = f"{info.get('turn')}:{info.get('currentPlayer')}"
    if main_action_counts.get(key, 0) < max_main_actions_per_turn:
        return None
    return find_action_index(info, "end_turn")


def remember_main_action(info: Dict[str, Any], action_index: int, main_action_counts: Dict[str, int]) -> None:
    if info.get("waitingForStep") != "main_phase":
        return
    legal_actions = info.get("legalActions")
    selected = legal_actions[action_index] if isinstance(legal_actions, list) and 0 <= action_index < len(legal_actions) else None
    if isinstance(selected, dict) and selected.get("step") == "end_turn":
        return
    key = f"{info.get('turn')}:{info.get('currentPlayer')}"
    main_action_counts[key] = main_action_counts.get(key, 0) + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a masked policy checkpoint for cyber-sim-engine.")
    parser.add_argument("--engine-root", default=str(default_engine_root()), help="Path to cyber-sim-engine repo root.")
    parser.add_argument("--deck1", default="decks/DEV-TEST-001.deck", help="Deck 1 path, relative to engine root unless absolute.")
    parser.add_argument("--deck2", default="decks/DEV-TEST-002.deck", help="Deck 2 path, relative to engine root unless absolute.")
    parser.add_argument("--checkpoint", default="rl/checkpoints/masked_policy_best.pt", help="Policy checkpoint path, relative to engine root unless absolute.")
    parser.add_argument("--node", default="node", help="Node executable path.")
    parser.add_argument("--first", default=None, choices=[None, "p1", "p2"], help="Force first player.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    parser.add_argument("--episodes", type=int, default=1000, help="Episodes per matchup.")
    parser.add_argument("--turn-cap", type=int, default=200, help="Engine turn cap.")
    parser.add_argument("--max-actions", type=int, default=128, help="Fixed action mask size.")
    parser.add_argument("--max-steps-per-episode", type=int, default=5000, help="Python-side safety step cap.")
    parser.add_argument("--max-main-actions-per-turn", type=int, default=12, help="Force end turn after this many non-end main actions per player-turn.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--stochastic", action="store_true", help="Sample from the policy instead of taking argmax.")
    parser.add_argument("--matchups", default="policy_vs_random,random_vs_policy", help="Comma-separated matchup names or all.")
    return parser.parse_args()


def parse_matchups(value: str) -> List[Tuple[str, str, str]]:
    if value.strip().lower() == "all":
        return list(DEFAULT_MATCHUPS)
    out: List[Tuple[str, str, str]] = []
    for raw in value.split(","):
        name = raw.strip()
        if not name:
            continue
        predefined = next((m for m in DEFAULT_MATCHUPS if m[0] == name), None)
        if predefined:
            out.append(predefined)
            continue
        if "_vs_" not in name:
            raise ValueError(f"Invalid matchup {name!r}. Expected p1Agent_vs_p2Agent.")
        p1, p2 = name.split("_vs_", 1)
        out.append((name, p1, p2))
    if not out:
        raise ValueError("No matchups selected")
    return out


def make_env(args: argparse.Namespace, engine_root: Path) -> CyberSimGymEnv:
    return CyberSimGymEnv(
        engine_root=str(engine_root),
        deck1=resolve_path(args.deck1, engine_root),
        deck2=resolve_path(args.deck2, engine_root),
        node_path=args.node,
        first_player=args.first,
        seed=args.seed,
        turn_cap=args.turn_cap,
        max_actions=args.max_actions,
        include_legal_actions=True,
    )


def make_named_agent(name: str, policy_agent: TorchPolicyAgent) -> BaseAgent:
    key = name.strip().lower().replace("-", "_")
    if key == "policy":
        return policy_agent  # type: ignore[return-value]
    return make_agent(key)


def run_matchup(
    *,
    name: str,
    p1_name: str,
    p2_name: str,
    policy_agent: TorchPolicyAgent,
    args: argparse.Namespace,
    engine_root: Path,
) -> Dict[str, Any]:
    p1_agent = make_named_agent(p1_name, policy_agent)
    p2_agent = make_named_agent(p2_name, policy_agent)
    rng = random.Random(args.seed + abs(hash(name)) % 1_000_000)
    stats: Dict[str, Any] = {
        "ok": True,
        "name": name,
        "p1Agent": getattr(p1_agent, "name", p1_name),
        "p2Agent": getattr(p2_agent, "name", p2_name),
        "episodes": args.episodes,
        "p1Wins": 0,
        "p2Wins": 0,
        "noWinner": 0,
        "errors": 0,
        "badObservations": 0,
        "badActionMasks": 0,
        "truncated": 0,
        "maxStepLimit": 0,
        "totalSteps": 0,
        "totalTurns": 0,
        "firstError": None,
        "firstBadObservation": None,
        "firstBadActionMask": None,
    }

    env: Optional[CyberSimGymEnv] = None
    try:
        env = make_env(args, engine_root)
        expected_size = env.observation_size

        for episode in range(args.episodes):
            try:
                obs, info = env.reset(seed=args.seed + episode)
                main_action_counts: Dict[str, int] = {}
                done = False
                steps = 0

                while not done and steps < args.max_steps_per_episode:
                    obs_error = validate_observation(obs, expected_size)
                    if obs_error:
                        stats["badObservations"] += 1
                        if stats["firstBadObservation"] is None:
                            stats["firstBadObservation"] = {"episode": episode, "error": obs_error}
                        raise RuntimeError(obs_error)

                    mask_error = validate_action_mask(info, args.max_actions)
                    if mask_error:
                        stats["badActionMasks"] += 1
                        if stats["firstBadActionMask"] is None:
                            stats["firstBadActionMask"] = {"episode": episode, "error": mask_error}
                        raise RuntimeError(mask_error)

                    forced = force_end_turn_if_needed(info, main_action_counts, args.max_main_actions_per_turn)
                    if forced is not None:
                        action_index = forced
                    else:
                        current_pid = info.get("currentPlayer") or info.get("activePlayer") or "p1"
                        agent = p1_agent if current_pid == "p1" else p2_agent
                        action_index = agent.select_action(obs, info, rng)

                    remember_main_action(info, action_index, main_action_counts)
                    obs, _reward, terminated, truncated, info = env.step(action_index)
                    steps += 1
                    done = bool(terminated or truncated)

                if steps >= args.max_steps_per_episode and not done:
                    stats["maxStepLimit"] += 1
                    raise RuntimeError("max_steps_per_episode")

                if bool(info.get("raw_response_done")) and not info.get("winner"):
                    stats["truncated"] += 1

                winner = info.get("winner")
                if winner == "p1":
                    stats["p1Wins"] += 1
                elif winner == "p2":
                    stats["p2Wins"] += 1
                else:
                    stats["noWinner"] += 1

                stats["totalSteps"] += steps
                stats["totalTurns"] += int(info.get("turn") or 0)

            except Exception as exc:
                stats["errors"] += 1
                stats["ok"] = False
                if stats["firstError"] is None:
                    stats["firstError"] = {"episode": episode, "error": str(exc)}
    finally:
        if env is not None:
            env.close()

    played = max(1, stats["p1Wins"] + stats["p2Wins"] + stats["noWinner"])
    stats["p1WinRate"] = round(stats["p1Wins"] / played, 4)
    stats["p2WinRate"] = round(stats["p2Wins"] / played, 4)
    stats["avgSteps"] = round(stats["totalSteps"] / max(1, args.episodes), 2)
    stats["avgTurns"] = round(stats["totalTurns"] / max(1, args.episodes), 2)
    stats["ok"] = stats["errors"] == 0 and stats["badObservations"] == 0 and stats["badActionMasks"] == 0
    return stats


def main() -> None:
    args = parse_args()
    engine_root = Path(args.engine_root).resolve()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = engine_root / checkpoint_path

    device = torch.device(args.device)
    model, metadata = load_checkpoint(checkpoint_path, device=device)
    model.eval()
    policy_agent = TorchPolicyAgent(model, device=device, deterministic=not args.stochastic)

    matchups = parse_matchups(args.matchups)
    results = {
        "ok": True,
        "checkpoint": str(checkpoint_path),
        "metadata": metadata,
        "episodesPerMatchup": args.episodes,
        "seed": args.seed,
        "matchups": {},
    }

    for name, p1_name, p2_name in matchups:
        result = run_matchup(
            name=name,
            p1_name=p1_name,
            p2_name=p2_name,
            policy_agent=policy_agent,
            args=args,
            engine_root=engine_root,
        )
        results["matchups"][name] = result
        if not result.get("ok"):
            results["ok"] = False

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
