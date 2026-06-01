#!/usr/bin/env python3
"""
Evaluate a saved masked PPO checkpoint.

This wraps the Step 6 policy evaluator and only changes the default checkpoint
path to rl/checkpoints/masked_ppo_best.pt.
"""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


def main() -> None:
    if "--checkpoint" not in sys.argv:
        sys.argv.extend(["--checkpoint", "rl/checkpoints/masked_ppo_best.pt"])
    from evaluate_masked_policy import main as evaluate_main  # noqa: WPS433
    evaluate_main()


if __name__ == "__main__":
    main()
