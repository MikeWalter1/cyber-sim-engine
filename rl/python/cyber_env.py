"""
Gymnasium-style Python bridge for cyber-sim-engine.

This file talks to rl/node_env_server.js over newline-delimited JSON. It is meant
for local experimentation first, not high-throughput training yet.

Basic usage from the engine repo root:

    from rl.python.cyber_env import CyberSimGymEnv, sample_legal_action

    env = CyberSimGymEnv(
        deck1="decks/DEV-TEST-001.deck",
        deck2="decks/DEV-TEST-002.deck",
    )
    obs, info = env.reset(seed=42)
    action = sample_legal_action(info)
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    np = None  # type: ignore

try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    gym = None  # type: ignore
    spaces = None  # type: ignore


def _repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[2]


def _as_abs_path(path_like: Optional[str], base: Path) -> Optional[str]:
    if path_like is None:
        return None
    p = Path(path_like)
    if not p.is_absolute():
        p = base / p
    return str(p.resolve())


def _finite_sequence(values: Iterable[Any]) -> bool:
    try:
        for value in values:
            if not math.isfinite(float(value)):
                return False
        return True
    except Exception:
        return False


def _to_observation(values: List[float]):
    if np is not None:
        return np.asarray(values, dtype=np.float32)
    return [float(v) for v in values]


def _mask_indices(action_mask: Iterable[Any]) -> List[int]:
    out: List[int] = []
    for index, value in enumerate(action_mask):
        if int(value) == 1:
            out.append(index)
    return out


def sample_legal_action(info: Dict[str, Any], rng: Optional[random.Random] = None) -> int:
    """Sample one valid action index from an info dict returned by reset/step."""
    rng = rng or random
    indices = _mask_indices(info.get("action_mask", []))
    if not indices:
        raise RuntimeError("No legal actions available in info['action_mask']")
    return rng.choice(indices)


class _BaseEnv(object):
    pass


if gym is not None:
    _GymBase = gym.Env
else:
    _GymBase = _BaseEnv


class CyberSimGymEnv(_GymBase):
    """
    Gymnasium-style environment for cyber-sim-engine.

    The action is an integer index into the current state's legal action list.
    The current action mask is provided as info["action_mask"].

    Reward convention for this first bridge:
      - non-terminal step: 0
      - terminal win for the player who just acted: +1
      - terminal loss for the player who just acted: -1
      - truncation / no winner: 0

    This lets one symmetric policy control both seats for smoke testing. Later
    training code can wrap this further for fixed-seat training versus an
    opponent policy.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        engine_root: Optional[str] = None,
        deck1: Optional[str] = None,
        deck2: Optional[str] = None,
        node_path: str = "node",
        server_path: Optional[str] = None,
        first_player: Optional[str] = None,
        seed: Optional[int] = None,
        turn_cap: int = 200,
        max_actions: int = 128,
        include_legal_actions: bool = False,
        startup_timeout_s: float = 10.0,
    ) -> None:
        super().__init__()

        self.engine_root = Path(engine_root).resolve() if engine_root else _repo_root_from_this_file()
        self.deck1 = _as_abs_path(deck1 or "decks/DEV-TEST-001.deck", self.engine_root)
        self.deck2 = _as_abs_path(deck2 or "decks/DEV-TEST-002.deck", self.engine_root)
        self.node_path = node_path
        self.server_path = Path(server_path).resolve() if server_path else self.engine_root / "rl" / "node_env_server.js"
        self.first_player = first_player
        self.initial_seed = seed
        self.turn_cap = int(turn_cap)
        self.max_actions = int(max_actions)
        self.include_legal_actions = bool(include_legal_actions)
        self.startup_timeout_s = float(startup_timeout_s)

        self._proc: Optional[subprocess.Popen[str]] = None
        self._next_id = 1
        self._closed = False
        self.observation_size: Optional[int] = None
        self.last_info: Dict[str, Any] = {}

        self._start_server()
        pong = self._request({"cmd": "ping"})
        self.observation_size = int(pong["observationSize"])

        if spaces is not None and np is not None:
            self.action_space = spaces.Discrete(self.max_actions)
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.observation_size,),
                dtype=np.float32,
            )
        else:
            self.action_space = None
            self.observation_space = None

    def _start_server(self) -> None:
        if not self.server_path.exists():
            raise FileNotFoundError(f"Node server not found: {self.server_path}")

        cmd = [
            self.node_path,
            str(self.server_path),
            "--engine-root",
            str(self.engine_root),
            "--deck1",
            str(self.deck1),
            "--deck2",
            str(self.deck2),
            "--turn-cap",
            str(self.turn_cap),
            "--max-actions",
            str(self.max_actions),
        ]

        if self.first_player:
            cmd.extend(["--first", self.first_player])
        if self.initial_seed is not None:
            cmd.extend(["--seed", str(int(self.initial_seed))])
        if self.include_legal_actions:
            cmd.append("--include-legal-actions")

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self.engine_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

    def _next_request_id(self) -> int:
        request_id = self._next_id
        self._next_id += 1
        return request_id

    def _read_stderr_tail(self) -> str:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return ""
        # Avoid blocking on a live process. Only read stderr after it exited.
        if proc.poll() is None:
            return ""
        try:
            return proc.stderr.read()[-4000:]
        except Exception:
            return ""

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._closed:
            raise RuntimeError("Environment is closed")

        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise RuntimeError("Node server process is not running")

        if proc.poll() is not None:
            err = self._read_stderr_tail()
            raise RuntimeError(f"Node server exited with code {proc.returncode}. {err}")

        request_id = self._next_request_id()
        payload = dict(payload)
        payload["id"] = request_id

        try:
            proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            proc.stdin.flush()
        except BrokenPipeError as exc:
            err = self._read_stderr_tail()
            raise RuntimeError(f"Node server pipe closed. {err}") from exc

        line = proc.stdout.readline()
        if not line:
            err = self._read_stderr_tail()
            raise RuntimeError(f"Node server returned no response. {err}")

        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON from Node server: {line[:500]}") from exc

        if response.get("id") != request_id:
            raise RuntimeError(f"Mismatched response id: expected {request_id}, got {response.get('id')}")

        if not response.get("ok"):
            raise RuntimeError(response.get("error") or "Unknown Node server error")

        return response

    def _convert_response(self, response: Dict[str, Any]) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        observation_values = response.get("observation")
        if not isinstance(observation_values, list):
            raise RuntimeError("Node response missing observation list")
        if self.observation_size is not None and len(observation_values) != self.observation_size:
            raise RuntimeError(f"Observation length {len(observation_values)} != expected {self.observation_size}")
        if not _finite_sequence(observation_values):
            raise RuntimeError("Observation contains non-finite values")

        action_mask = response.get("actionMask")
        if not isinstance(action_mask, list):
            raise RuntimeError("Node response missing actionMask list")
        if len(action_mask) != self.max_actions:
            raise RuntimeError(f"Action mask length {len(action_mask)} != max_actions {self.max_actions}")
        if any(int(v) not in (0, 1) for v in action_mask):
            raise RuntimeError("Action mask contains values other than 0/1")

        info = dict(response.get("info") or {})
        info["action_mask"] = [int(v) for v in action_mask]
        info["raw_response_done"] = bool(response.get("done"))

        self.last_info = info

        observation = _to_observation([float(v) for v in observation_values])
        reward = float(response.get("reward", 0.0))
        terminated = bool(response.get("terminated", False))
        truncated = bool(response.get("truncated", False))
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        command: Dict[str, Any] = {"cmd": "reset"}
        if seed is not None:
            command["seed"] = int(seed)
        elif self.initial_seed is not None:
            command["seed"] = int(self.initial_seed)

        options = options or {}
        first_player = options.get("firstPlayer", options.get("first_player", self.first_player))
        if first_player:
            command["firstPlayer"] = first_player

        response = self._request(command)
        observation, _reward, _terminated, _truncated, info = self._convert_response(response)
        return observation, info

    def step(self, action: int):
        try:
            action_index = int(action)
        except Exception as exc:
            raise ValueError(f"Action must be convertible to int, got {action!r}") from exc

        response = self._request({"cmd": "step", "actionIndex": action_index})
        return self._convert_response(response)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        proc = self._proc
        if proc is None:
            return

        try:
            if proc.poll() is None and proc.stdin is not None:
                request_id = self._next_request_id()
                proc.stdin.write(json.dumps({"cmd": "close", "id": request_id}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass

        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def __enter__(self) -> "CyberSimGymEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


__all__ = ["CyberSimGymEnv", "sample_legal_action"]
