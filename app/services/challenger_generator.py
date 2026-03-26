from __future__ import annotations

import json
import random
from copy import deepcopy
from typing import Any

from app.config import ResearchConfig
from shared.state import default_strategy_parameters


def _parameter_signature(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True)


def _mutated_parameter_names(config: ResearchConfig, rng: random.Random) -> list[str]:
    all_names = list(config.mutation_ranges.keys())
    mutation_count = min(max(2, len(all_names) // 6), 4)
    return rng.sample(all_names, k=mutation_count)


def generate_challenger_variants(
    champion: dict[str, Any],
    config: ResearchConfig,
    existing_parameters: set[str],
    cycle_number: int,
) -> list[dict[str, Any]]:
    """Generate safe parameterized challenger variants without rewriting code."""
    base_parameters = deepcopy(champion.get("parameters") or default_strategy_parameters())
    seed = f"{champion.get('id', 'champion')}:{cycle_number}:{len(existing_parameters)}"
    rng = random.Random(seed)
    challengers: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(config.challengers_per_cycle * 8, 12)

    while len(challengers) < config.challengers_per_cycle and attempts < max_attempts:
        attempts += 1
        mutated = deepcopy(base_parameters)
        for field_name in _mutated_parameter_names(config, rng):
            options = list(config.mutation_ranges.get(field_name, []))
            if not options:
                continue
            current_value = mutated.get(field_name)
            alternative_options = [value for value in options if value != current_value]
            if alternative_options:
                mutated[field_name] = rng.choice(alternative_options)
        if mutated.get("ema_short_len", 0) >= mutated.get("ema_long_len", 0):
            mutated["ema_short_len"] = min(config.mutation_ranges["ema_short_len"])
            mutated["ema_long_len"] = max(config.mutation_ranges["ema_long_len"])
        signature = _parameter_signature(mutated)
        if signature in existing_parameters:
            continue
        existing_parameters.add(signature)
        challengers.append(mutated)
    return challengers
