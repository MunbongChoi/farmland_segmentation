"""YAML loading, base-file merging, and CLI override helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copied ``base`` mapping."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and recursively merge files listed in ``_base_``."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        current = yaml.safe_load(stream) or {}
    base_names = current.pop("_base_", [])
    if isinstance(base_names, str):
        base_names = [base_names]
    merged: dict[str, Any] = {}
    for base_name in base_names:
        merged = deep_merge(merged, load_config(config_path.parent / base_name))
    return deep_merge(merged, current)


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``section.key=value`` strings, parsing values as YAML scalars."""
    result = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"CLI override는 key=value 형식이어야 합니다: {item}")
        dotted_key, raw_value = item.split("=", 1)
        keys = dotted_key.split(".")
        cursor = result
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = yaml.safe_load(raw_value)
    return result


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Write the resolved configuration for reproducibility."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)

