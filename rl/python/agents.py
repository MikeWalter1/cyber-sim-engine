"""
Baseline agents for the cyber-sim-engine RL bridge.

These agents intentionally choose from the current state's legal action list.
They are meant for environment validation and baseline comparison before PPO or
other learning algorithms are added.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _valid_indices(info: Dict[str, Any]) -> List[int]:
    mask = info.get("action_mask", [])
    out: List[int] = []
    for index, value in enumerate(mask):
        try:
            if int(value) == 1:
                out.append(index)
        except Exception:
            continue
    return out


def _legal_actions(info: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions = info.get("legalActions")
    if not isinstance(actions, list):
        return []
    return [a for a in actions if isinstance(a, dict)]


def legal_action_pairs(info: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    """Return selectable ``(index, action)`` pairs for the current state."""
    actions = _legal_actions(info)
    pairs: List[Tuple[int, Dict[str, Any]]] = []
    for index in _valid_indices(info):
        if 0 <= index < len(actions):
            pairs.append((index, actions[index]))
    return pairs


def find_action_index(info: Dict[str, Any], step: str) -> Optional[int]:
    """Find the first valid action index with the given engine action step."""
    for index, action in legal_action_pairs(info):
        if action.get("step") == step:
            return index
    return None


class BaseAgent:
    """Small interface used by ``evaluate_agents.py``."""

    name = "base"

    def select_action(self, observation: Any, info: Dict[str, Any], rng: random.Random) -> int:
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """Uniformly sample one currently legal action index."""

    name = "random"

    def select_action(self, observation: Any, info: Dict[str, Any], rng: random.Random) -> int:
        valid = _valid_indices(info)
        if not valid:
            raise RuntimeError("RandomAgent: no legal action indices in action_mask")
        return rng.choice(valid)


class EndTurnAgent(BaseAgent):
    """Mostly useful as a debugging baseline."""

    name = "end_turn"

    def select_action(self, observation: Any, info: Dict[str, Any], rng: random.Random) -> int:
        index = find_action_index(info, "end_turn")
        if index is not None:
            return index
        return RandomAgent().select_action(observation, info, rng)


class HeuristicAgent(BaseAgent):
    """
    Simple action-object heuristic.

    It does not inspect the full board. It only uses the current legal action
    objects exposed by the Node bridge. That keeps this baseline independent of
    hidden implementation details and gives a stable sanity check before neural
    training.
    """

    name = "heuristic"

    def __init__(self, random_tiebreak: bool = False) -> None:
        self.random_tiebreak = bool(random_tiebreak)

    def select_action(self, observation: Any, info: Dict[str, Any], rng: random.Random) -> int:
        pairs = legal_action_pairs(info)
        if not pairs:
            return RandomAgent().select_action(observation, info, rng)

        scored = [(self.score_action(action, info), index, action) for index, action in pairs]
        best_score = max(score for score, _index, _action in scored)
        best = [(index, action) for score, index, action in scored if score == best_score]

        if self.random_tiebreak and len(best) > 1:
            return rng.choice(best)[0]
        return min(index for index, _action in best)

    def score_action(self, action: Dict[str, Any], info: Dict[str, Any]) -> float:
        step = action.get("step")
        waiting_step = info.get("waitingForStep")

        if step == "choose_gig_die":
            # Higher-side dice tend to produce more street cred and stronger plays.
            return 1000.0 + float(action.get("sides") or 0)

        if step == "choose_gig_to_steal":
            # Any listed steal choice is legal; prefer stealing more if that ever varies.
            return 950.0 + len(action.get("iids") or [])

        if step == "effect_choice_response":
            return self._score_effect_choice(action)

        if waiting_step == "attacker_interrupt_step":
            if step == "activate_asset_spend":
                return 800.0
            if step == "pass_attacker_interrupt":
                return 100.0

        if waiting_step == "defensive_step":
            if step == "blocker":
                return 900.0
            if step == "activate_asset_spend":
                return 850.0
            if step == "call_legend_defensive":
                return 650.0
            if step == "pass_defensive":
                return 100.0

        if waiting_step == "main_phase":
            return self._score_main_phase_action(action)

        # Safe fallback for any future legal-action type.
        return 0.0

    def _score_main_phase_action(self, action: Dict[str, Any]) -> float:
        step = action.get("step")

        if step == "activate_anytime_spend":
            return 920.0

        if step == "declare_attack":
            target = action.get("target") if isinstance(action.get("target"), dict) else {}
            if target.get("kind") == "gigs":
                return 900.0
            if target.get("kind") == "unit":
                return 850.0
            return 825.0

        if step == "play_card":
            # Gear actions have equip_to; make units/programs slightly preferred when tied.
            return 780.0 if "equip_to" not in action else 760.0

        if step == "call_legend":
            return 700.0

        if step == "sell_card":
            return 620.0

        if step == "tap_resource":
            return 500.0

        if step == "end_turn":
            return 100.0

        if step == "untap_resource":
            # Avoid random tap/untap loops.
            return -100.0

        return 0.0

    def _score_effect_choice(self, action: Dict[str, Any]) -> float:
        response = action.get("response")
        if not isinstance(response, dict):
            return 0.0

        if "accept" in response:
            # Prefer using optional effects. This is a simple baseline, not perfect play.
            return 760.0 if response.get("accept") is True else 700.0

        if "amount" in response:
            try:
                return 740.0 + float(response.get("amount") or 0)
            except Exception:
                return 740.0

        if "selected_iids" in response:
            selected = response.get("selected_iids")
            return 730.0 + (len(selected) if isinstance(selected, list) else 0)

        if "iid" in response:
            # Pick concrete targets over optional null skips.
            return 720.0 if response.get("iid") is not None else 650.0

        return 700.0


def make_agent(name: str) -> BaseAgent:
    key = str(name).strip().lower().replace("-", "_")
    if key == "random":
        return RandomAgent()
    if key in ("heuristic", "simple", "greedy"):
        return HeuristicAgent()
    if key in ("end", "end_turn", "pass"):
        return EndTurnAgent()
    raise ValueError(f"Unknown agent: {name!r}. Available: random, heuristic, end_turn")


__all__ = [
    "BaseAgent",
    "RandomAgent",
    "HeuristicAgent",
    "EndTurnAgent",
    "make_agent",
    "legal_action_pairs",
    "find_action_index",
]
