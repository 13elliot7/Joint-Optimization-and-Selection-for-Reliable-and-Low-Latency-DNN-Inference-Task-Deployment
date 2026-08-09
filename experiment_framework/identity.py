from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any


def _normalize(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _normalize(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return _normalize(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, set):
        return sorted((_normalize(item) for item in value), key=repr)
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("experiment identity cannot contain NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported experiment identity value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        _normalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_hash(value: Any, length: int = 16) -> str:
    if length <= 0 or length > 64:
        raise ValueError("hash length must be in [1, 64]")
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def derive_seed(base_seed: int, stream: str, algorithm: str | None = None) -> int:
    if base_seed < 0 or not stream:
        raise ValueError("base_seed must be non-negative and stream must be non-empty")
    material = {"base_seed": base_seed, "stream": stream, "algorithm": algorithm}
    return int.from_bytes(hashlib.sha256(canonical_json(material).encode()).digest()[:8], "big")


def task_id(
    *,
    experiment_layer: str,
    protocol_version: str,
    suite: str,
    scenario: Any,
    algorithm: str,
    repeat: int,
    seed: int,
) -> str:
    scenario_digest = stable_hash(scenario)
    return (
        f"{experiment_layer}:{protocol_version}:{suite}:{scenario_digest}:"
        f"{algorithm}:r{repeat}:s{seed}"
    )
