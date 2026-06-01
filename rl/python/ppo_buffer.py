"""
Rollout buffer utilities for masked PPO training.

The environment has a dynamic legal action list represented as a fixed-size
mask. Each transition stores the observation, action mask, selected action,
old log-probability, value estimate, reward, and done flag for one learning
policy decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


@dataclass
class PpoTransition:
    observation: List[float]
    action_mask: List[int]
    action: int
    old_log_prob: float
    value: float
    reward: float = 0.0
    done: bool = False


class PpoRolloutBuffer:
    def __init__(self) -> None:
        self.transitions: List[PpoTransition] = []

    def __len__(self) -> int:
        return len(self.transitions)

    def clear(self) -> None:
        self.transitions.clear()

    def extend(self, items: Iterable[PpoTransition]) -> None:
        self.transitions.extend(items)

    def add_episode(self, episode: Sequence[PpoTransition], terminal_reward: float) -> None:
        """Add one finished episode and place the terminal reward on its last policy decision."""
        if not episode:
            return
        copied: List[PpoTransition] = []
        for i, transition in enumerate(episode):
            copied.append(PpoTransition(
                observation=list(transition.observation),
                action_mask=list(transition.action_mask),
                action=int(transition.action),
                old_log_prob=float(transition.old_log_prob),
                value=float(transition.value),
                reward=float(terminal_reward) if i == len(episode) - 1 else 0.0,
                done=i == len(episode) - 1,
            ))
        self.transitions.extend(copied)

    def to_tensors(self, *, gamma: float, gae_lambda: float, device: Any) -> Dict[str, Any]:
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for PpoRolloutBuffer.to_tensors")
        if not self.transitions:
            raise RuntimeError("Cannot tensorize an empty PPO rollout buffer")

        observations = torch.as_tensor([t.observation for t in self.transitions], dtype=torch.float32, device=device)
        action_masks = torch.as_tensor([t.action_mask for t in self.transitions], dtype=torch.float32, device=device)
        actions = torch.as_tensor([t.action for t in self.transitions], dtype=torch.long, device=device)
        old_log_probs = torch.as_tensor([t.old_log_prob for t in self.transitions], dtype=torch.float32, device=device)
        values = torch.as_tensor([t.value for t in self.transitions], dtype=torch.float32, device=device)
        rewards = [float(t.reward) for t in self.transitions]
        dones = [bool(t.done) for t in self.transitions]

        advantages = compute_gae(rewards, values.detach().cpu().tolist(), dones, gamma=float(gamma), gae_lambda=float(gae_lambda))
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        returns_t = advantages_t + values

        return {
            "observations": observations,
            "action_masks": action_masks,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "old_values": values,
            "advantages": advantages_t,
            "returns": returns_t,
        }


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    *,
    gamma: float,
    gae_lambda: float,
) -> List[float]:
    """
    Compute generalized advantage estimates over policy-decision steps.

    Episodes are currently complete before insertion, so the last transition has
    done=True and a bootstrap value of 0. This also works for concatenated full
    episodes because each terminal done flag resets the recursion.
    """
    n = len(rewards)
    if len(values) != n or len(dones) != n:
        raise ValueError("rewards, values, and dones must have the same length")

    advantages = [0.0 for _ in range(n)]
    gae = 0.0
    for i in range(n - 1, -1, -1):
        if dones[i]:
            next_non_terminal = 0.0
            next_value = 0.0
        else:
            next_non_terminal = 1.0
            next_value = float(values[i + 1]) if i + 1 < n else 0.0
        delta = float(rewards[i]) + gamma * next_value * next_non_terminal - float(values[i])
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[i] = gae
    return advantages


__all__ = ["PpoTransition", "PpoRolloutBuffer", "compute_gae"]
