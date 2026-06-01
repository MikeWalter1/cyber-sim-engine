#!/usr/bin/env python3
"""
Evaluate simple baseline agents through the CyberSimGymEnv bridge.

Run from the cyber-sim-engine repo root:

    py rl/python/evaluate_agents.py \
      --deck1 decks/DEV-TEST-001.deck \
      --deck2 decks/DEV-TEST-002.deck \
      --episodes 1000 \
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from agents import BaseAgent, find_action_index, make_agent  # noqa: E402
from cyber_env import CyberSimGymEnv  # noqa: E402


DEFAULT_MATCHUPS = [
    ("random_vs_random", "random", "random"),
    ("heuristic_vs_random", "heuristic", "random"),
    ("random_vs_heuristic", "random", "heuristic"),
]


def default_engine_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline agents for cyber-sim-engine RL environment.")
    parser.add_argument("--engine-root", default=str(default_engine_root()), help="Path to cyber-sim-engine repo root.")
    parser.add_argument("--deck1", default="decks/DEV-TEST-001.deck", help="Deck 1 path, relative to engine root unless absolute.")
    parser.add_argument("--deck2", default="decks/DEV-TEST-002.deck", help="Deck 2 path, relative to engine root unless absolute.")
    parser.add_argument("--node", default="node", help="Node executable path.")
    parser.add_argument("--first", default=None, choices=[None, "p1", "p2"], help="Force first player.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    parser.add_argument("--episodes", type=int, default=1000, help="Episodes per matchup.")
    parser.add_argument("--turn-cap", type=int, default=200, help="Engine turn cap.")
    parser.add_argument("--max-actions", type=int, default=128, help="Fixed action mask size.")
    parser.add_argument("--max-steps-per-episode", type=int, default=5000, help="Python-side safety step cap.")
    parser.add_argument("--max-main-actions-per-turn", type=int, default=12, help="Force end turn after this many non-end main actions per player-turn.")
    parser.add_argument(
        "--matchups",
        default="all",
        help="Comma-separated matchup list. Use all, random_vs_random, heuristic_vs_random, random_vs_heuristic, or p1Agent_vs_p2Agent.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress to stderr.")
    return parser.parse_args()


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
    if active != int(info.get("maskedActionCount", active)):
        return "action_mask active count does not match info['maskedActionCount']"
    if bool(info.get("actionMaskOverflow")):
        return "actionMaskOverflow is true"
    if "legalActions" not in info:
        return "info['legalActions'] missing; evaluator requires include_legal_actions=True"
    return None


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
        p1_agent, p2_agent = name.split("_vs_", 1)
        out.append((name, p1_agent, p2_agent))

    if not out:
        raise ValueError("No matchups selected")
    return out


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


def choose_agent_for_current_player(info: Dict[str, Any], p1_agent: BaseAgent, p2_agent: BaseAgent) -> BaseAgent:
    pid = info.get("currentPlayer") or info.get("activePlayer") or "p1"
    return p1_agent if pid == "p1" else p2_agent


def run_matchup(
    *,
    name: str,
    p1_agent_name: str,
    p2_agent_name: str,
    args: argparse.Namespace,
    engine_root: Path,
) -> Dict[str, Any]:
    p1_agent = make_agent(p1_agent_name)
    p2_agent = make_agent(p2_agent_name)
    rng = random.Random((args.seed or 0) + abs(hash(name)) % 1_000_000)

    stats: Dict[str, Any] = {
        "ok": True,
        "name": name,
        "p1Agent": p1_agent.name,
        "p2Agent": p2_agent.name,
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
        "observationSize": None,
        "maxActions": args.max_actions,
        "firstError": None,
        "firstBadObservation": None,
        "firstBadActionMask": None,
        "errorBuckets": {},
    }

    env: Optional[CyberSimGymEnv] = None
    try:
        env = CyberSimGymEnv(
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
        stats["observationSize"] = env.observation_size

        for episode in range(args.episodes):
            episode_seed = (args.seed + episode) if args.seed is not None else None
            main_action_counts: Dict[str, int] = {}
            final_info: Dict[str, Any] = {}
            episode_error: Optional[BaseException] = None
            steps = 0
            obs: Any = []
            info: Dict[str, Any] = {}

            try:
                obs, info = env.reset(seed=episode_seed)
            except BaseException as exc:  # noqa: BLE001
                episode_error = exc

            if episode_error is None:
                obs_error = validate_observation(obs, env.observation_size)
                if obs_error:
                    stats["badObservations"] += 1
                    if not stats["firstBadObservation"]:
                        stats["firstBadObservation"] = {"episode": episode + 1, "seed": episode_seed, "error": obs_error}
                mask_error = validate_action_mask(info, args.max_actions)
                if mask_error:
                    stats["badActionMasks"] += 1
                    if not stats["firstBadActionMask"]:
                        stats["firstBadActionMask"] = {"episode": episode + 1, "seed": episode_seed, "error": mask_error, "info": info}

            terminated = False
            truncated = False

            while episode_error is None and not (terminated or truncated) and steps < args.max_steps_per_episode:
                try:
                    forced = force_end_turn_if_needed(info, main_action_counts, args.max_main_actions_per_turn)
                    if forced is not None:
                        action_index = forced
                    else:
                        agent = choose_agent_for_current_player(info, p1_agent, p2_agent)
                        action_index = agent.select_action(obs, info, rng)

                    remember_main_action(info, action_index, main_action_counts)
                    obs, _reward, terminated, truncated, info = env.step(action_index)
                    final_info = info
                    steps += 1
                except BaseException as exc:  # noqa: BLE001
                    episode_error = exc
                    break

                obs_error = validate_observation(obs, env.observation_size)
                if obs_error:
                    stats["badObservations"] += 1
                    if not stats["firstBadObservation"]:
                        stats["firstBadObservation"] = {
                            "episode": episode + 1,
                            "seed": episode_seed,
                            "step": steps,
                            "error": obs_error,
                        }
                    break

                mask_error = validate_action_mask(info, args.max_actions)
                if mask_error:
                    stats["badActionMasks"] += 1
                    if not stats["firstBadActionMask"]:
                        stats["firstBadActionMask"] = {
                            "episode": episode + 1,
                            "seed": episode_seed,
                            "step": steps,
                            "error": mask_error,
                            "info": info,
                        }
                    break

            if episode_error is not None:
                stats["errors"] += 1
                message = str(episode_error)
                stats["errorBuckets"][message] = stats["errorBuckets"].get(message, 0) + 1
                if not stats["firstError"]:
                    stats["firstError"] = {
                        "episode": episode + 1,
                        "seed": episode_seed,
                        "step": steps,
                        "error": message,
                    }
            elif steps >= args.max_steps_per_episode and not (terminated or truncated):
                stats["maxStepLimit"] += 1
                stats["noWinner"] += 1
            elif truncated:
                stats["truncated"] += 1
                stats["noWinner"] += 1
            else:
                winner = final_info.get("winner") or info.get("winner")
                if winner == "p1":
                    stats["p1Wins"] += 1
                elif winner == "p2":
                    stats["p2Wins"] += 1
                else:
                    stats["noWinner"] += 1

            stats["totalSteps"] += steps
            stats["totalTurns"] += int((final_info or info).get("turn") or 0)

            if args.verbose and ((episode + 1) % max(1, args.episodes // 20) == 0 or episode == args.episodes - 1):
                print(f"{name}: {episode + 1}/{args.episodes}", file=sys.stderr)

    finally:
        if env is not None:
            env.close()

    completed = max(1, stats["episodes"] - stats["errors"] - stats["badObservations"] - stats["badActionMasks"])
    decisive = max(1, stats["p1Wins"] + stats["p2Wins"])
    stats["p1WinRate"] = round(stats["p1Wins"] / decisive, 4)
    stats["p2WinRate"] = round(stats["p2Wins"] / decisive, 4)
    stats["avgSteps"] = round(stats["totalSteps"] / stats["episodes"], 2) if stats["episodes"] else 0
    stats["avgTurns"] = round(stats["totalTurns"] / stats["episodes"], 2) if stats["episodes"] else 0
    stats["ok"] = (
        stats["errors"] == 0
        and stats["badObservations"] == 0
        and stats["badActionMasks"] == 0
        and stats["truncated"] == 0
        and stats["maxStepLimit"] == 0
    )
    return stats


def main() -> int:
    args = parse_args()
    engine_root = Path(args.engine_root).resolve()
    matchups = parse_matchups(args.matchups)

    results: Dict[str, Any] = {
        "ok": True,
        "episodesPerMatchup": args.episodes,
        "seed": args.seed,
        "matchups": {},
    }

    for name, p1_agent, p2_agent in matchups:
        result = run_matchup(
            name=name,
            p1_agent_name=p1_agent,
            p2_agent_name=p2_agent,
            args=args,
            engine_root=engine_root,
        )
        results["matchups"][name] = result
        if not result.get("ok"):
            results["ok"] = False

    print(json.dumps(results, indent=2))
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
