from __future__ import annotations

import json
from pathlib import Path


FORMAL_SUITES = (
    "search_quality",
    "load",
    "algorithm_baseline",
    "robustness",
    "scale",
)


def load_plan(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    suites = tuple(payload.get("suites", ()))
    if suites != FORMAL_SUITES:
        raise ValueError(
            "formal experiment plan must contain exactly the five converged suites "
            f"in order: {', '.join(FORMAL_SUITES)}"
        )
    if int(payload.get("repeats", 0)) <= 0:
        raise ValueError("formal experiment repeats must be positive")
    if int(payload.get("base_seed", -1)) < 0:
        raise ValueError("formal experiment base_seed must be non-negative")
    return payload
