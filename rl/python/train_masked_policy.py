#!/usr/bin/env python3
"""
Train a first masked policy-gradient agent for cyber-sim-engine.

This is deliberately a compact baseline, not full PPO yet. It trains one seat
(p1 by default) against a non-learning opponent using the existing Python bridge.

Run from the cyber-sim-engine repo root:

    py rl/python/train_masked_policy.py \
      --deck1 decks/DEV-TEST-001.deck \
      --deck2 decks/DEV-TEST-002.deck \
      --episodes 2000 \
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    print(json.dumps({
        "ok": False,
        "error": "PyTorch is required for Step 6.",
        "install": "py -m pip install torch numpy",
        "detail": str(exc),
    }, indent=2))
    raise SystemExit(1)

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from agents import make_agent, find_action_index  # noqa: E402
from cyber_env import CyberSimGymEnv  # noqa: E402
from masked_policy import (  # noqa: E402
    MaskedActorCritic,
    sample_masked_action,
    save_checkpoint,
)


@dataclass
class Transition:
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor


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


def discounted_terminal_returns(reward: float, count: int, gamma: float) -> List[float]:
    if count <= 0:
        return []
    gamma = float(gamma)
    return [float(reward) * (gamma ** (count - 1 - i)) for i in range(count)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a masked policy-gradient baseline for cyber-sim-engine.")
    parser.add_argument("--engine-root", default=str(default_engine_root()), help="Path to cyber-sim-engine repo root.")
    parser.add_argument("--deck1", default="decks/DEV-TEST-001.deck", help="Deck 1 path, relative to engine root unless absolute.")
    parser.add_argument("--deck2", default="decks/DEV-TEST-002.deck", help="Deck 2 path, relative to engine root unless absolute.")
    parser.add_argument("--node", default="node", help="Node executable path.")
    parser.add_argument("--first", default=None, choices=[None, "p1", "p2"], help="Force first player.")
    parser.add_argument("--train-pid", default="p1", choices=["p1", "p2"], help="Seat controlled by the learning policy.")
    parser.add_argument("--opponent", default="random", choices=["random", "heuristic", "end_turn"], help="Opponent policy.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    parser.add_argument("--episodes", type=int, default=2000, help="Training episodes.")
    parser.add_argument("--batch-episodes", type=int, default=16, help="Policy update cadence in episodes.")
    parser.add_argument("--turn-cap", type=int, default=200, help="Engine turn cap.")
    parser.add_argument("--max-actions", type=int, default=128, help="Fixed action mask size.")
    parser.add_argument("--max-steps-per-episode", type=int, default=5000, help="Python-side safety step cap.")
    parser.add_argument("--max-main-actions-per-turn", type=int, default=12, help="Force end turn after this many non-end main actions per player-turn.")
    parser.add_argument("--hidden-size", type=int, default=256, help="MLP hidden size.")
    parser.add_argument("--hidden-layers", type=int, default=2, help="MLP hidden layer count.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount applied across learning-player decisions inside an episode.")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient.")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--eval-interval", type=int, default=100, help="Evaluate every N training episodes. Use 0 to disable.")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Episodes per evaluation pass.")
    parser.add_argument("--eval-mode", default="both", choices=["deterministic", "stochastic", "both"], help="Evaluation mode used during training. deterministic = argmax, stochastic = sample, both = report both and select best by stochastic win rate.")
    parser.add_argument("--checkpoint-dir", default="rl/checkpoints", help="Checkpoint directory, relative to engine root unless absolute.")
    parser.add_argument("--checkpoint-name", default="masked_policy_latest.pt", help="Latest checkpoint filename.")
    parser.add_argument("--best-name", default="masked_policy_best.pt", help="Best checkpoint filename.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--progress-jsonl", action="store_true", help="Print progress records as JSON lines instead of compact stderr updates.")
    return parser.parse_args()


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


def evaluate_policy(
    *,
    model: MaskedActorCritic,
    args: argparse.Namespace,
    engine_root: Path,
    device: torch.device,
    episodes: int,
    base_seed: int,
    deterministic: bool,
) -> Dict[str, Any]:
    opponent = make_agent(args.opponent)
    rng = random.Random(base_seed + 91_337)
    stats: Dict[str, Any] = {
        "episodes": episodes,
        "mode": "deterministic" if deterministic else "stochastic",
        "trainPid": args.train_pid,
        "opponent": opponent.name,
        "wins": 0,
        "losses": 0,
        "noWinner": 0,
        "errors": 0,
        "totalSteps": 0,
        "totalTurns": 0,
        "firstError": None,
    }

    env: Optional[CyberSimGymEnv] = None
    try:
        env = make_env(args, engine_root)
        expected_size = env.observation_size

        for episode in range(episodes):
            try:
                obs, info = env.reset(seed=base_seed + episode)
                main_action_counts: Dict[str, int] = {}
                done = False
                steps = 0

                while not done and steps < args.max_steps_per_episode:
                    obs_error = validate_observation(obs, expected_size)
                    mask_error = validate_action_mask(info, args.max_actions)
                    if obs_error or mask_error:
                        raise RuntimeError(obs_error or mask_error)

                    forced = force_end_turn_if_needed(info, main_action_counts, args.max_main_actions_per_turn)
                    current_pid = info.get("currentPlayer") or info.get("activePlayer") or "p1"
                    if forced is not None:
                        action_index = forced
                    elif current_pid == args.train_pid:
                        action_index = int(sample_masked_action(
                            model,
                            obs,
                            info.get("action_mask", []),
                            device=device,
                            deterministic=deterministic,
                        ).action_index)
                    else:
                        action_index = opponent.select_action(obs, info, rng)

                    remember_main_action(info, action_index, main_action_counts)
                    obs, _reward, terminated, truncated, info = env.step(action_index)
                    steps += 1
                    done = bool(terminated or truncated)

                if steps >= args.max_steps_per_episode and not done:
                    stats["errors"] += 1
                    if stats["firstError"] is None:
                        stats["firstError"] = {"episode": episode, "error": "max_steps_per_episode"}
                    continue

                winner = info.get("winner")
                if winner == args.train_pid:
                    stats["wins"] += 1
                elif winner in ("p1", "p2"):
                    stats["losses"] += 1
                else:
                    stats["noWinner"] += 1
                stats["totalSteps"] += steps
                stats["totalTurns"] += int(info.get("turn") or 0)

            except Exception as exc:
                stats["errors"] += 1
                if stats["firstError"] is None:
                    stats["firstError"] = {"episode": episode, "error": str(exc)}
    finally:
        if env is not None:
            env.close()

    played = max(1, stats["wins"] + stats["losses"] + stats["noWinner"])
    stats["winRate"] = round(stats["wins"] / played, 4)
    stats["avgSteps"] = round(stats["totalSteps"] / max(1, episodes), 2)
    stats["avgTurns"] = round(stats["totalTurns"] / max(1, episodes), 2)
    stats["ok"] = stats["errors"] == 0
    return stats


def evaluate_policy_suite(
    *,
    model: MaskedActorCritic,
    args: argparse.Namespace,
    engine_root: Path,
    device: torch.device,
    episodes: int,
    base_seed: int,
) -> Dict[str, Any]:
    """Run one or both evaluation modes and return a stable result object."""
    out: Dict[str, Any] = {}

    if args.eval_mode in ("deterministic", "both"):
        out["deterministic"] = evaluate_policy(
            model=model,
            args=args,
            engine_root=engine_root,
            device=device,
            episodes=episodes,
            base_seed=base_seed,
            deterministic=True,
        )

    if args.eval_mode in ("stochastic", "both"):
        out["stochastic"] = evaluate_policy(
            model=model,
            args=args,
            engine_root=engine_root,
            device=device,
            episodes=episodes,
            base_seed=base_seed + 50_000,
            deterministic=False,
        )

    if args.eval_mode == "deterministic":
        primary = out["deterministic"]
    elif args.eval_mode == "stochastic":
        primary = out["stochastic"]
    else:
        primary = out.get("stochastic") or out.get("deterministic") or {}

    return {
        "mode": args.eval_mode,
        "primary": primary,
        **out,
    }


def main() -> None:
    args = parse_args()
    engine_root = Path(args.engine_root).resolve()
    device = torch.device(args.device)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = engine_root / checkpoint_dir
    latest_path = checkpoint_dir / args.checkpoint_name
    best_path = checkpoint_dir / args.best_name

    env: Optional[CyberSimGymEnv] = None
    start_time = time.time()

    stats: Dict[str, Any] = {
        "ok": True,
        "episodes": args.episodes,
        "trainPid": args.train_pid,
        "opponent": args.opponent,
        "seed": args.seed,
        "device": str(device),
        "wins": 0,
        "losses": 0,
        "noWinner": 0,
        "errors": 0,
        "policyActions": 0,
        "opponentActions": 0,
        "updates": 0,
        "totalSteps": 0,
        "totalTurns": 0,
        "firstError": None,
        "latestCheckpoint": str(latest_path),
        "bestCheckpoint": str(best_path),
        "bestEvalWinRate": None,
        "lastLoss": None,
        "lastPolicyLoss": None,
        "lastValueLoss": None,
        "lastEntropy": None,
        "lastEval": None,
    }

    batch_transitions: List[Transition] = []
    batch_returns: List[float] = []
    recent_rewards: List[float] = []

    try:
        env = make_env(args, engine_root)
        observation_size = int(env.observation_size or 0)
        model = MaskedActorCritic(
            observation_size=observation_size,
            max_actions=args.max_actions,
            hidden_size=args.hidden_size,
            hidden_layers=args.hidden_layers,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        opponent = make_agent(args.opponent)

        stats["observationSize"] = observation_size
        stats["maxActions"] = args.max_actions

        for episode in range(1, args.episodes + 1):
            episode_transitions: List[Transition] = []
            main_action_counts: Dict[str, int] = {}
            episode_seed = args.seed + episode - 1

            try:
                obs, info = env.reset(seed=episode_seed)
                done = False
                steps = 0

                while not done and steps < args.max_steps_per_episode:
                    obs_error = validate_observation(obs, observation_size)
                    mask_error = validate_action_mask(info, args.max_actions)
                    if obs_error or mask_error:
                        raise RuntimeError(obs_error or mask_error)

                    forced = force_end_turn_if_needed(info, main_action_counts, args.max_main_actions_per_turn)
                    current_pid = info.get("currentPlayer") or info.get("activePlayer") or "p1"

                    if forced is not None:
                        action_index = forced
                    elif current_pid == args.train_pid:
                        decision = sample_masked_action(
                            model,
                            obs,
                            info.get("action_mask", []),
                            device=device,
                            deterministic=False,
                        )
                        action_index = decision.action_index
                        episode_transitions.append(Transition(decision.log_prob, decision.value, decision.entropy))
                        stats["policyActions"] += 1
                    else:
                        action_index = opponent.select_action(obs, info, rng)
                        stats["opponentActions"] += 1

                    remember_main_action(info, action_index, main_action_counts)
                    obs, _env_reward, terminated, truncated, info = env.step(action_index)
                    steps += 1
                    done = bool(terminated or truncated)

                if steps >= args.max_steps_per_episode and not done:
                    raise RuntimeError("max_steps_per_episode")

                winner = info.get("winner")
                if winner == args.train_pid:
                    terminal_reward = 1.0
                    stats["wins"] += 1
                elif winner in ("p1", "p2"):
                    terminal_reward = -1.0
                    stats["losses"] += 1
                else:
                    terminal_reward = 0.0
                    stats["noWinner"] += 1

                stats["totalSteps"] += steps
                stats["totalTurns"] += int(info.get("turn") or 0)
                recent_rewards.append(terminal_reward)
                if len(recent_rewards) > 100:
                    recent_rewards.pop(0)

                returns = discounted_terminal_returns(terminal_reward, len(episode_transitions), args.gamma)
                batch_transitions.extend(episode_transitions)
                batch_returns.extend(returns)

                should_update = episode % args.batch_episodes == 0 or episode == args.episodes
                if should_update and batch_transitions:
                    log_probs = torch.stack([t.log_prob for t in batch_transitions])
                    values = torch.stack([t.value for t in batch_transitions]).float()
                    entropies = torch.stack([t.entropy for t in batch_transitions]).float()
                    returns_t = torch.as_tensor(batch_returns, dtype=torch.float32, device=device)

                    advantages = returns_t - values.detach()
                    policy_loss = -(log_probs * advantages).mean()
                    value_loss = F.mse_loss(values, returns_t)
                    entropy = entropies.mean()
                    loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                    stats["updates"] += 1
                    stats["lastLoss"] = round(float(loss.detach().cpu()), 6)
                    stats["lastPolicyLoss"] = round(float(policy_loss.detach().cpu()), 6)
                    stats["lastValueLoss"] = round(float(value_loss.detach().cpu()), 6)
                    stats["lastEntropy"] = round(float(entropy.detach().cpu()), 6)

                    batch_transitions.clear()
                    batch_returns.clear()

                if args.eval_interval > 0 and episode % args.eval_interval == 0:
                    eval_stats = evaluate_policy_suite(
                        model=model,
                        args=args,
                        engine_root=engine_root,
                        device=device,
                        episodes=args.eval_episodes,
                        base_seed=args.seed + 1_000_000 + episode,
                    )
                    stats["lastEval"] = eval_stats
                    primary_eval = eval_stats.get("primary") if isinstance(eval_stats, dict) else {}
                    win_rate = float((primary_eval or {}).get("winRate") or 0.0)
                    if stats["bestEvalWinRate"] is None or win_rate > float(stats["bestEvalWinRate"]):
                        stats["bestEvalWinRate"] = win_rate
                        save_checkpoint(
                            best_path,
                            model,
                            optimizer,
                            episode=episode,
                            eval=eval_stats,
                            args=vars(args),
                        )

                    det_win = eval_stats.get("deterministic", {}).get("winRate") if isinstance(eval_stats, dict) else None
                    sto_win = eval_stats.get("stochastic", {}).get("winRate") if isinstance(eval_stats, dict) else None

                    if args.progress_jsonl:
                        print(json.dumps({
                            "episode": episode,
                            "recentAvgReward": round(sum(recent_rewards) / max(1, len(recent_rewards)), 4),
                            "wins": stats["wins"],
                            "losses": stats["losses"],
                            "updates": stats["updates"],
                            "eval": eval_stats,
                        }, separators=(",", ":")), flush=True)
                    else:
                        eval_bits = []
                        if det_win is not None:
                            eval_bits.append(f"eval_det={float(det_win):.3f}")
                        if sto_win is not None:
                            eval_bits.append(f"eval_sto={float(sto_win):.3f}")
                        if not eval_bits:
                            eval_bits.append(f"eval_win={win_rate:.3f}")
                        print(
                            f"episode {episode}/{args.episodes} "
                            f"recent_reward={sum(recent_rewards)/max(1,len(recent_rewards)):.3f} "
                            f"{' '.join(eval_bits)} "
                            f"loss={stats['lastLoss']}",
                            file=sys.stderr,
                            flush=True,
                        )

                if episode % max(1, args.batch_episodes * 10) == 0 or episode == args.episodes:
                    save_checkpoint(
                        latest_path,
                        model,
                        optimizer,
                        episode=episode,
                        stats=stats,
                        args=vars(args),
                    )

            except Exception as exc:
                stats["errors"] += 1
                stats["ok"] = False
                if stats["firstError"] is None:
                    stats["firstError"] = {"episode": episode, "seed": episode_seed, "error": str(exc)}
                # Continue so a rare episode issue is reported with context rather than losing all progress.
                continue

        save_checkpoint(
            latest_path,
            model,
            optimizer,
            episode=args.episodes,
            stats=stats,
            args=vars(args),
        )

    finally:
        if env is not None:
            env.close()

    played = max(1, stats["wins"] + stats["losses"] + stats["noWinner"])
    stats["winRate"] = round(stats["wins"] / played, 4)
    stats["avgSteps"] = round(stats["totalSteps"] / max(1, played), 2)
    stats["avgTurns"] = round(stats["totalTurns"] / max(1, played), 2)
    stats["elapsedSec"] = round(time.time() - start_time, 2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
