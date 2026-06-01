"""
Masked PyTorch policy utilities for cyber-sim-engine.

This module is intentionally small and dependency-light apart from PyTorch.
It uses the fixed observation vector and fixed action mask produced by
rl/python/cyber_env.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


VERY_NEGATIVE = -1.0e9


class MaskedActorCritic(nn.Module):
    """Simple MLP with a masked policy head and a scalar value head."""

    def __init__(
        self,
        observation_size: int,
        max_actions: int,
        hidden_size: int = 256,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.observation_size = int(observation_size)
        self.max_actions = int(max_actions)
        self.hidden_size = int(hidden_size)
        self.hidden_layers = int(hidden_layers)

        layers = []
        in_features = self.observation_size
        for _ in range(max(1, self.hidden_layers)):
            layers.append(nn.Linear(in_features, self.hidden_size))
            layers.append(nn.ReLU())
            in_features = self.hidden_size

        self.body = nn.Sequential(*layers)
        self.policy_head = nn.Linear(in_features, self.max_actions)
        self.value_head = nn.Linear(in_features, 1)

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        x = self.body(observation.float())
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value


def as_tensor_2d(values: Any, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=dtype, device=device)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def mask_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Replace illegal-action logits with a very negative value."""
    if action_mask.dim() == 1:
        action_mask = action_mask.unsqueeze(0)
    action_mask = action_mask.to(device=logits.device)
    legal = action_mask > 0.5
    if legal.shape != logits.shape:
        raise ValueError(f"Action mask shape {tuple(legal.shape)} does not match logits {tuple(logits.shape)}")
    return torch.where(legal, logits, torch.full_like(logits, VERY_NEGATIVE))


@dataclass
class PolicyDecision:
    action_index: int
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor


def sample_masked_action(
    model: MaskedActorCritic,
    observation: Any,
    action_mask: Any,
    *,
    device: torch.device,
    deterministic: bool = False,
) -> PolicyDecision:
    obs_t = as_tensor_2d(observation, device=device, dtype=torch.float32)
    mask_t = as_tensor_2d(action_mask, device=device, dtype=torch.float32)

    logits, value = model(obs_t)
    masked = mask_logits(logits, mask_t)

    if deterministic:
        action = torch.argmax(masked, dim=-1)
        dist = Categorical(logits=masked)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
    else:
        dist = Categorical(logits=masked)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

    return PolicyDecision(
        action_index=int(action.item()),
        log_prob=log_prob.squeeze(0),
        value=value.squeeze(0),
        entropy=entropy.squeeze(0),
    )


@torch.no_grad()
def choose_masked_action(
    model: MaskedActorCritic,
    observation: Any,
    action_mask: Any,
    *,
    device: torch.device,
    deterministic: bool = True,
) -> int:
    decision = sample_masked_action(
        model,
        observation,
        action_mask,
        device=device,
        deterministic=deterministic,
    )
    return decision.action_index


def save_checkpoint(
    path: str | Path,
    model: MaskedActorCritic,
    optimizer: Optional[torch.optim.Optimizer] = None,
    **metadata: Any,
) -> None:
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "observation_size": model.observation_size,
        "max_actions": model.max_actions,
        "hidden_size": model.hidden_size,
        "hidden_layers": model.hidden_layers,
        "metadata": metadata,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[MaskedActorCritic, Dict[str, Any]]:
    payload = torch.load(str(path), map_location=device)
    model = MaskedActorCritic(
        observation_size=int(payload["observation_size"]),
        max_actions=int(payload["max_actions"]),
        hidden_size=int(payload.get("hidden_size", 256)),
        hidden_layers=int(payload.get("hidden_layers", 2)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return model, dict(payload.get("metadata") or {})


class TorchPolicyAgent:
    """Agent wrapper compatible with the Step 5 evaluator loop."""

    name = "policy"

    def __init__(self, model: MaskedActorCritic, *, device: torch.device, deterministic: bool = True) -> None:
        self.model = model
        self.device = device
        self.deterministic = bool(deterministic)

    def select_action(self, observation: Any, info: Dict[str, Any], rng: Any = None) -> int:
        return choose_masked_action(
            self.model,
            observation,
            info.get("action_mask", []),
            device=self.device,
            deterministic=self.deterministic,
        )


__all__ = [
    "MaskedActorCritic",
    "PolicyDecision",
    "TorchPolicyAgent",
    "as_tensor_2d",
    "mask_logits",
    "sample_masked_action",
    "choose_masked_action",
    "save_checkpoint",
    "load_checkpoint",
]
