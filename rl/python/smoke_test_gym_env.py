#!/usr/bin/env python3
"""
Smoke test for the Python Gymnasium-style bridge.

Run from the cyber-sim-engine repo root:

    python rl/python/smoke_test_gym_env.py \
      --deck1 decks/DEV-TEST-001.deck \
      --deck2 decks/DEV-TEST-002.deck \
      --episodes 100 \
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cyber_env import CyberSimGymEnv, sample_legal_action  # noqa: E402


def default_engine_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the cyber-sim-engine Python bridge.")
    parser.add_argument("--engine-root", default=str(default_engine_root()), help="Path to cyber-sim-engine repo root.")
    parser.add_argument("--deck1", default="decks/DEV-TEST-001.deck", help="Deck 1 path, relative to engine root unless absolute.")
    parser.add_argument("--deck2", default="decks/DEV-TEST-002.deck", help="Deck 2 path, relative to engine root unless absolute.")
    parser.add_argument("--node", default="node", help="Node executable path.")
    parser.add_argument("--first", default=None, choices=[None, "p1", "p2"], help="Force first player.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run.")
    parser.add_argument("--turn-cap", type=int, default=200, help="Engine turn cap.")
    parser.add_argument("--max-actions", type=int, default=128, help="Fixed action mask size.")
    parser.add_argument("--max-steps-per-episode", type=int, default=5000, help="Hard Python-side safety step cap.")
    parser.add_argument("--max-main-actions-per-turn", type=int, default=12, help="Force end turn after this many non-end main actions per player-turn.")
    parser.add_argument("--include-legal-actions", action="store_true", help="Ask Node server to include legal action objects in info for debugging.")
    parser.add_argument("--verbose", action="store_true", help="Print one line per episode to stderr.")
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
    bad = [v for v in mask if int(v) not in (0, 1)]
    if bad:
        return "action_mask contains non-binary entries"
    if sum(int(v) for v in mask) != int(info.get("maskedActionCount", sum(int(v) for v in mask))):
        return "action_mask active count does not match info['maskedActionCount']"
    if bool(info.get("actionMaskOverflow")):
        return "actionMaskOverflow is true"
    return None


def should_force_end_turn(info: Dict[str, Any], main_action_counts: Dict[str, int], max_main_actions_per_turn: int) -> bool:
    if info.get("waitingForStep") != "main_phase":
        return False
    key = f"{info.get('turn')}:{info.get('currentPlayer')}"
    return main_action_counts.get(key, 0) >= max_main_actions_per_turn


def remember_main_action(info: Dict[str, Any], action_index: int, main_action_counts: Dict[str, int]) -> None:
    if info.get("waitingForStep") != "main_phase":
        return

    selected = info.get("selectedAction")
    if isinstance(selected, dict) and selected.get("step") == "end_turn":
        return

    # Without legalActions, action index 0 might still be end_turn depending on the
    # generator, but the forced-end-turn safety is only a loop guard; exact counting
    # is not important for correctness.
    key = f"{info.get('turn')}:{info.get('currentPlayer')}"
    main_action_counts[key] = main_action_counts.get(key, 0) + 1


def choose_action(info: Dict[str, Any], rng: random.Random, max_main_actions_per_turn: int, main_action_counts: Dict[str, int]) -> int:
    mask = info.get("action_mask", [])
    valid = [i for i, value in enumerate(mask) if int(value) == 1]
    if not valid:
        raise RuntimeError("No valid action indices in action mask")

    if should_force_end_turn(info, main_action_counts, max_main_actions_per_turn):
        # Legal actions are contiguous in Step 3. If legalActions are included,
        # prefer the explicit end_turn action. Otherwise fall back to a random
        # action; the main purpose is to avoid pathological random loops.
        legal_actions = info.get("legalActions")
        if isinstance(legal_actions, list):
            for index, action in enumerate(legal_actions):
                if index in valid and isinstance(action, dict) and action.get("step") == "end_turn":
                    return index

    # Encourage games to progress when no legalActions are included.
    if info.get("waitingForStep") == "main_phase" and rng.random() < 0.20:
        legal_actions = info.get("legalActions")
        if isinstance(legal_actions, list):
            for index, action in enumerate(legal_actions):
                if index in valid and isinstance(action, dict) and action.get("step") == "end_turn":
                    return index

    return rng.choice(valid)


def main() -> int:
    args = parse_args()
    engine_root = Path(args.engine_root).resolve()
    rng = random.Random(args.seed)

    stats: Dict[str, Any] = {
        "ok": True,
        "episodes": args.episodes,
        "p1Wins": 0,
        "p2Wins": 0,
        "noWinner": 0,
        "errors": 0,
        "badObservations": 0,
        "badActionMasks": 0,
        "truncated": 0,
        "maxStepLimit": 0,
        "totalReward": 0.0,
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
            include_legal_actions=args.include_legal_actions,
        )
        stats["observationSize"] = env.observation_size

        for episode in range(args.episodes):
            episode_seed = args.seed + episode if args.seed is not None else None
            main_action_counts: Dict[str, int] = {}
            episode_error: Optional[BaseException] = None
            final_info: Dict[str, Any] = {}
            steps = 0
            total_reward = 0.0

            try:
                obs, info = env.reset(seed=episode_seed)
            except BaseException as exc:  # noqa: BLE001
                episode_error = exc
                obs, info = [], {}

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
                    action = choose_action(info, rng, args.max_main_actions_per_turn, main_action_counts)
                    obs, reward, terminated, truncated, info = env.step(action)
                    steps += 1
                    total_reward += float(reward)
                    final_info = info
                    remember_main_action(info, action, main_action_counts)
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
            stats["totalReward"] += total_reward

            if args.verbose:
                print(
                    f"episode {episode + 1}/{args.episodes}: "
                    f"winner={(final_info or info).get('winner')} "
                    f"turn={(final_info or info).get('turn')} steps={steps} "
                    f"reward={total_reward:.1f}"
                    f"{f' error={episode_error}' if episode_error else ''}",
                    file=sys.stderr,
                )

    finally:
        if env is not None:
            env.close()

    stats["avgSteps"] = round(stats["totalSteps"] / args.episodes, 2) if args.episodes else 0
    stats["avgTurns"] = round(stats["totalTurns"] / args.episodes, 2) if args.episodes else 0
    stats["avgReward"] = round(stats["totalReward"] / args.episodes, 4) if args.episodes else 0
    stats["ok"] = (
        stats["errors"] == 0
        and stats["badObservations"] == 0
        and stats["badActionMasks"] == 0
        and stats["truncated"] == 0
        and stats["maxStepLimit"] == 0
    )

    print(json.dumps(stats, indent=2))
    return 0 if stats["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
